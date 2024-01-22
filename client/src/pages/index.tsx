import * as ImageProvider from "@/constants/ImageProvider"
export default function Page() {
    return (
        <div className="w-full h-auto">
            <div className="bg-slate-700 w-full h-auto">
                <img
                    src={ImageProvider.VIETNAM_AIRLINE}
                    alt='airline'
                    className="w-full h-96 object-cover shadow-md"
                />
            </div>
            <div></div>
        </div>
    )
}